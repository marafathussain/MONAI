import wget

# COVID Negative: it-1
url = 'https://b2drop.bsc.es/index.php/s/BIMCV-COVID19-Negative/download'
out_dir = '/home/marafath/scratch/bimcv/covid_neg'
wget.download(url, out=out_dir)

# COVID Positive: it-1&2
url = 'https://b2drop.bsc.es/index.php/s/BIMCV-COVID19/download'
out_dir = '/home/marafath/scratch/bimcv/covid_pos'
wget.download(url, out=out_dir)

