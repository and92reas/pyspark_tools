from . import *

class Tools:

    '''
    contains a toolbox of simple useful functions that act as building blocks for different kinds of projects
    '''

    def show(p):
        '''
        *p*: matplotlib.pyplot object
        '''
        img = io.StringIO()
        p.savefig(img, format='svg')
        img.seek(0)
        print("%html <div " + img.getvalue() + "</div>")

    def print_sequence(seq):
        '''
        prints a list with each element appearing to a different row
        *seq*: sequence to be printed
        '''
        _ = list(map(lambda x: print(x), seq))
