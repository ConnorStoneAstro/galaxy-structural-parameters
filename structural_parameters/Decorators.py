import functools

class okPROBESerror(Exception):
    pass

def catch_errors(func):
    @functools.wraps(func)
    def wrapper_catch_errors(G, *args, **kwargs):
        try:
            return func(G, *args, **kwargs)
        except KeyError as e:
            if 'warnings' in G:
                G['warnings'].append(e)
            return G
        except okPROBESerror as e:
            if 'warnings' in G:
                G['warnings'].append(e)
            return G
            
    return wrapper_catch_errors

def all_appR(func):
    @functools.wraps(func)
    def wrapper_all_appR(G, *args, eval_at_R=None, eval_at_band=None, **kwargs):
        if not eval_at_R is None:
            return func(G, *args, eval_at_R=eval_at_R, eval_at_band=eval_at_band, **kwargs)
        for R in G['appR']:
            if "E|" in R:
                continue
            G = func(G, *args, eval_at_R=R.split(":")[0], eval_at_band=R.split(":")[1], **kwargs)
        return G
    return wrapper_all_appR

def all_bands(func):
    @functools.wraps(func)
    def wrapper_all_bands(G, *args, eval_in_band=None, **kwargs):
        if not eval_in_band is None:
            return func(G, *args, eval_in_band=eval_in_band, **kwargs)
        for b in list(G['photometry']):
            G = func(G, *args, eval_in_band = b, **kwargs)
        return G
    return wrapper_all_bands

