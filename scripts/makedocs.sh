if [ $(basename $(pwd)) = 'cvlab_toolbox' ]
    then
        sphinx-apidoc -f -o ./docs/source ./cvt
        sphinx-build ./docs ./docs/_build
        make html
    else
        echo Wrong directory
        echo Run in cvlab_toolbox

fi