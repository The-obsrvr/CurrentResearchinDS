import numpy as np
import pandas as pd
from alibi.utils.mapping import ohe_to_ord
from alibi.explainers import CounterFactualProto as CFP

class Counterfactual():
    
    def __init__(self, model, data, data_handler):
        '''
        initialize an explainer and fit it on the model and data
        '''
        self.data_handler = data_handler
        input_shape = (1, len(data.columns))
        feature_range = (np.array([[0] * len(self.data_handler.cat_features) +
                                   list(data[self.data_handler.num_features].min(axis=0))]).astype(np.float32),
                         np.array([[1] * len(self.data_handler.cat_features) + 
                                   list(data[self.data_handler.num_features].max(axis=0))]).astype(np.float32))

        self.predict_fn = lambda x: model.predict_proba(x)
        self.cf = CFP(self.predict_fn, input_shape, use_kdtree=True, theta=10.,
                      cat_vars=data_handler.cat_vars_ohe, ohe=True, max_iterations=1000,
                      feature_range=feature_range)
        self.cf.fit(np.array(data), d_type='abdm')

  
    def describe_instance(self, eps=1e-4):

        target_classes = {0: "no", 1:"yes"}

        print('Original instance: {}  -- proba: {}'.format(target_classes[self.explanation.orig_class],
                                                           self.explanation.orig_proba[0]))
        print('Counterfactual instance: {}  -- proba: {}'.format(target_classes[self.explanation.cf['class']],
                                                                 self.explanation.cf['proba'][0]))
        print('\nCounterfactual perturbations...')
        print('\nCategorical:')

        # Convert ohe to ord
        X_orig_ord = ohe_to_ord(self.X, self.data_handler.cat_vars_ohe)[0]
        X_cf_ord = ohe_to_ord(self.explanation.cf['X'], 
                              self.data_handler.cat_vars_ohe)[0]

        # Check variations between the original and counterfactual
        delta_cat = {}
        for i, (_, v) in enumerate(self.data_handler.category_map.items()):
            cat_orig = v[int(X_orig_ord[0, i])]
            cat_cf = v[int(X_cf_ord[0, i])]
            if cat_orig != cat_cf:
                delta_cat[self.data_handler.cat_features[i]] = [cat_orig, cat_cf]
        if delta_cat:
            for k, v in delta_cat.items():
                print('{}: {}  -->   {}'.format(k, v[0], v[1]))

        print('\nNumerical:')
        delta_num = X_cf_ord[0, -8:] - X_orig_ord[0, -8:]

        for i in range(delta_num.shape[0]):
            if np.abs(delta_num[i]) > eps:
                print('{}: {:.2f}  -->   {:.2f}'.format(self.data_handler.num_features[i],
                                                X_orig_ord[0,i+len(self.data_handler.cat_features)],
                                                X_cf_ord[0,i+len(self.data_handler.cat_features)]))


    def define_counterfactuals(self, X):
        '''
        define the possible counterfactuals for the given instance

        :X: Dataframe row for the data instance
        '''
        self.X = np.array(X).reshape((1, len(X)))
        self.explanation = self.cf.explain(self.X, k=2)

        if self.explanation.cf == None:
            return None

        self.describe_instance()

        return pd.DataFrame(self.explanation.cf['X'], 
                            columns=self.data_handler.column_names)
