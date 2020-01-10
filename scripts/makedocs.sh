if [ $(basename $(pwd)) = 'cvlab_toolbox' ]
    then
        sphinx-apidoc -f -o ./docs ./docs/_build
        sphinx-build ./docs ./docs/_build
        make html
    else
        echo Wrong directory
        echo Run in cvlab_toolbox

fi