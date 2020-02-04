
## Coding styles
- Follow `PEP8` as much as possible
  - [English](https://www.python.org/dev/peps/pep-0008/)
  - [日本語](http://pep8-ja.readthedocs.io/ja/latest/)
- Write a description as **NumPy Style Python Docstring**
  - [numpydoc docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html)


  ```python
  def PCA(X, whiten = False):
    '''
      apply PCA
      components, explained_variance = PCA(X)

      Parameters
      ----------
      X: ndarray, shape (n_samples, n_features)
        matrix of input vectors

      whiten: boolean
        if it is True, the data is treated as whitened
        on each dimensions (average is 0 and variance is 1)

      Returns
      -------
      components: ndarray, shape (n_features, n_features)
        the normalized component vectors

      explained_variance: ndarray, shape (n_features)
        the variance of each vectors
    '''

    ...
  ```

## Contribution rules
1. Make a pull request
2. Ask some lab members to review the code
3. when all agreements are taken, ask any admin member to merge it

## Guidelines for writing documentation

Below is an excerpt from [Scikit learn docs](https://scikit-learn.org/stable/developers/contributing.html#guidelines-for-writing-documentation) with a few minor changes for emphasis. 

> It is important to keep a good compromise between mathematical and algorithmic details, and give intuition to the reader on what the algorithm does.

> Basically, to elaborate on the above, it is best to **always start with a small paragraph with a hand-waving explanation of what the method does to the data**. Then, it is very helpful to **point out why the feature is useful and when it should be used** - the latter also **including “big O” () complexities** of the algorithm, as opposed to just rules of thumb, as the latter can be very machine-dependent. If those complexities are not available, then rules of thumb may be provided instead.

> Secondly, a generated **figure from an example should then be included** to further provide some intuition.

> Next, **one or two small code examples** to show its use can be added.

> **Lastly, any math and equations, followed by references, can be added** to further the documentation. Not starting the documentation with the maths makes it more friendly towards users that are just interested in what the feature will do, as opposed to how it works “under the hood”.

## Mathematical Notation

I generally try to follow the [notation of the Deep Learning book](https://github.com/goodfeli/dlbook_notation) by Ian Goodfellow.

Below is a list of the notation used here.

**Numbers and Arrays**

|Symbol|Latex|Description  |
|:---:|:---:|:---:|
|`$a$`| $a$ | A scalar (integer or real)|
|$\va$||A vector|
|||A matrix|
|||A tensor|
|||Standard basis vector|
||||
||||

**Datasets and Distributions**

|Symbol|Latex|Description  |
|:---:|:---:|:---:|
|`$\mathbb{X}$`||A set of training examples|
|`$\mathbf{x}^{(i)}$`||The i-th example from a dataset|
|`$y^{(i)}$` or `$\mathbf{y}^{(i)}$`||The target associated with x for supervised learning|
|`$\mathbf{x}'$`||A new example from a test dataset|


