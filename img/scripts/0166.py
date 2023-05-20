from bing_image_downloader import downloader

downloader.download("pirate image portrait", limit=1000,  output_dir='dataset', adult_filter_off=False, force_replace=False, timeout=60, verbose=True)
