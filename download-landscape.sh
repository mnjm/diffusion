#!/usr/bin/env bash

outdir="./dataset-cache/landscapes"
tempzipfile="./landscape-pictures.zip"

curl -L -o $tempzipfile "https://www.kaggle.com/api/v1/datasets/download/arnaud58/landscape-pictures"

if [ $? -ne 0 ]; then
    echo "Error: Failed to download the dataset."
    exit 1
fi

mkdir -p $outdir

echo "Unzipping dataset to $outdir.."
unzip -q $tempzipfile -d $outdir

if [ $? -ne 0 ]; then
    echo "Error: Failed to unzip the dataset."
    exit 1
fi

rm $tempzipfile

# Process each JPG file
for file in $outdir/*.jpg; do
    if [[ "$file" =~ _\(([0-9]+)\)\.jpg$ ]]; then
        # File has a tag (like _(2).jpg)
        tag="${BASH_REMATCH[1]}"
        mkdir -p "$outdir/$tag"
        mv "$file" "$outdir/$tag/"
    else
        # File has no tag (simple .jpg)
        mkdir -p "$outdir/1"
        mv "$file" "$outdir/1/"
    fi
done

echo Done
