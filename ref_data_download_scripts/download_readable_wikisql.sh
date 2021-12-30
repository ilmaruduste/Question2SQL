mkdir wikisql
python -c "import gdown; gdown.download('https://drive.google.com/uc?export=download&id=1TBT9MnaGOUcQ3rCXo2KAXXUYEwkrY133', 'wikisql/wikisql.zip', quiet=False)"
cd wikisql
unzip wikisql.zip
rm wikisql.zip