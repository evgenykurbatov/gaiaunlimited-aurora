# Do not run this script until you know what you're doing!

exit 1

for d in "pl-*"; do
    echo - $d
    echo -- Remove auxiliaries
    rm -rf $d/__pycache__
    echo -- Remove figures
    rm -rf $d/fig/*
    echo -- Remove datasets
    rm $d/*.csv $d/*.pkl $d/*.npz
done
