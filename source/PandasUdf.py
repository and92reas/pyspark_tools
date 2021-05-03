from . import *

class PandasUdf():

    '''
    contains examples of pandas user defined functions
    '''

    @pandas_udf('double', PandasUDFType.SCALAR)
    def get_perc_of_digits(s):
        '''
        pandas_udf version of get_perc_of_digits
        *s*: pandas series
        '''
        str_len = s.str.len()
        return pd.Series(np.where(str_len == 0, 0, s.str.count("\d") / str_len))


    @pandas_udf('double', PandasUDFType.SCALAR)
    def get_perc_of_punc_symbols(s,
                                 punc_set = string.punctuation))):
        '''
        pandas_udf version of get_perc_of_punc_symbols
        *s*: pandas series
        *punc_set*: set containing the punctuation symbols to take into account
        '''
        str_len = s.str.len()
        return pd.Series(np.where(str_len == 0, 0, s.str.count("[{}]".format(punc_set)) / str_len))
