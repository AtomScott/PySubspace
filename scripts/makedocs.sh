if [ $(basename $(pwd)) = 'cvlab_toolbox' ]
    then
        sphinx-build ./docs _build
        # I think sphinx-build runs all the below as well.
        # sphinx-autogen ./docs/source/classes.rst
        # sphinx-apidoc -f -o ./docs/source ./cvt
        make -C docs html
    else
        echo Wrong directory
        echo Run in cvlab_toolbox

fi