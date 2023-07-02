import inspect
import equinox
from typing import Callable, Iterable

def compose(workflow: Iterable[Callable]) -> Callable:   
    def pipeline(*args, **kwargs): # *args are for grad, **kwargs are the rest
        res = dict([])

        for f in workflow:
            sig = inspect.signature(f)
            f_args = sig.parameters.keys()

            feed_args = False
            feed_kwargs = False

            arglist = []
            
            for arg in f_args:
                if not feed_args or not feed_kwargs:
                    if arg in kwargs.keys() and arg not in res.keys():
                        feed_kwargs = True
                        arglist.append(arg)
                    elif arg not in kwargs.keys() and arg not in res.keys():
                        feed_args = True
                    elif arg in kwargs.keys() and arg in res.keys():
                        raise Exception(f'the keyword \'{arg}\' is already specified in the workflow')
                else:
                    break

            f_kwargs = {k:kwargs[k] for k in arglist}

            if feed_args and feed_kwargs:
                res = f(*args, **res, **f_kwargs)
            elif feed_args and not feed_kwargs:
                res = f(*args, **res)
            elif not feed_args and feed_kwargs:
                res = f(**res, **f_kwargs)
            else:
                res = f(**res)

        return res
    
    # not really too helpful, since can't parse which of these are free params...
    workflow_pars = []
    for _i, f in enumerate(workflow):
        sig = inspect.signature(f)
        workflow_pars += list(sig.parameters.values())
    
    workflow_pars = sorted(workflow_pars, key=lambda x: 0 if x.default is inspect.Parameter.empty else 1)
    last_sig = inspect.signature(workflow[-1])
    an = last_sig.return_annotation
    pipeline.__signature__ = inspect.Signature(workflow_pars, return_annotation=an)
    
    return pipeline
            
  