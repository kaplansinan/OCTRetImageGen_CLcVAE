## for further details of the data source,check 
# https://data.mendeley.com/datasets/rscbjbr9sj/2?__hstc=25856994.0439b2c2a493ab64e4f4c39d5d679633.1660301993493.1660301993493.1660301993493.1

# Download data 
wget -c https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/5699a1d8-d1b6-45db-bb92-b61051445347/file_downloaded 

# Unzip file to "data"
tar -xzf file_downloaded -C data

# Remove "data.zip"
rm file_downloaded


echo "Data downloaded and organised."