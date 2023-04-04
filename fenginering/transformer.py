import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import ParameterGrid

from sklearn.pipeline import Pipeline, FeatureUnion


def param_grid(params):
    return list(ParameterGrid(params))

def sin_transformer(period):
    return FunctionTransformer(lambda x : np.sin(x / period*2*np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x : np.cos(x / period*2*np.pi))

def crossover(funct):
    return FunctionTransformer(funct)



class MakeIndicator:
    
    def __init__(self, data):
        self.data = data.copy()
        self.names = []
        self.steps = []
    
    def one_params(self, data, params, funct):
        df = pd.DataFrame()
        for param in params:
            df = pd.concat([df, funct(data, param)], axis = 1)
        return df
    
    def multi_params(self, data, params_grid, funct):
        df = pd.DataFrame()
        for param in params_grid:
            df = pd.concat([df, funct(data, **param)], axis = 1)
        return df
    
    def param_grid(self, params):
        return list(ParameterGrid(params))
    
    def add_colname(self, funct, params):
        for i, param in enumerate(params):
            if isinstance(param, dict):
                cols = funct(self.data, **param).columns.to_list()
            else:
                cols = funct(self.data, param).name
                cols = [cols]
            for j in range(len(cols)):
                cols[j] = cols[j] + "_" + str(i)
            self.names.extend(cols)
                
            
    def set_stransformer(self, funct, params):
        self.add_colname(funct, params)
        transformer = FunctionTransformer(self.one_params, kw_args = {"params" : params, "funct" : funct})
        self.add_transformer(name = funct.__name__, transformer = transformer)
        return transformer
        
    def set_mtransformer(self, funct, params):
        params = self.param_grid(params)
        self.add_colname(funct, params)
        transformer = FunctionTransformer(self.multi_params, kw_args = {"params_grid" : params, "funct" : funct})
        self.add_transformer(name = funct.__name__, transformer = transformer)
        return transformer
    
    def dropna_transformer(self):
        return FunctionTransformer(lambda x : pd.DataFrame(x, columns = self.names).dropna())
    
    def add_transformer(self, name, transformer):
        step = (name, transformer)
        self.steps.append(step)
    
    def pipeline(self):
        dropna = self.dropna_transformer()
        
        union = FeatureUnion(
            self.steps
        )
        pipe = Pipeline(
            [
                ("indicator", union),
                ("dropna",  dropna)
            ]
        )
        return pipe

