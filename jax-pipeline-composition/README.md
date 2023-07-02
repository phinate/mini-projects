# composing jax pipelines of functions

when i was initially coding in jax, i found that i was writing a lot of code that looked like this:

```python
def f(x):
    return x + 1

def g(x):
    return x * 2

def h(x):
    return x - 3

def pipeline(x):
    y = f(x)
    z = g(y)
    a = h(z)
    return a
```

to differentiate a chain of function calls wrt a value early in the pipeline, you end up making very long versions of this, and also have to feed in the specific kwargs for each function. this is a very common pattern in jax, and i wanted to find a way to make it easier to do. i came up with this:

```python
from melody import compose
pipeline = compose([f, g, h])
```

it's not really that good of an implementation since it forces the first arg to be the value you want to differentiate wrt, but it's a start. you also have to be really specific with the kwargs you pass in, and need to force the previous function to return a dict with the correct keys that correspond to the kwargs, i.e.

```python
def f(x):
    return {'y': x + 1}

def g(y):  # y here is the key in the dict returned by f
    return {'z': x * 2}

def h(z):
    return {'a': x - 3}
```

if you think this is interesting and you want a better version, we can talk about it more (or feel free to just make your own based on this).

