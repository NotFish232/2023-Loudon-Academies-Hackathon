// define required libraries (fs, req, scraper)
var fs = require('fs'), request = require('request');
var scrape = require('images-scraper');
const n = 1024;

//define download function
var download = function(uri, filename, callback){
  request.head(uri, function(err, res, body){
    console.log('content-type:', res.headers['content-type']);
    console.log('content-length:', res.headers['content-length']);

    request(uri).pipe(fs.createWriteStream(filename)).on('close', callback);
  });
};


//run scraping
const google = new scrape({
  puppeteer: {
    headless: false,
  },
});

(async () => {
  const results = await google.scrape('pirate image portrait', n);

  for(let i = 0; i < results.length; i++) {
    let obj = results[i];

    download(obj.url, "pirate_" + i + '.png', function(){
      console.log(i + "/" + results.length);
    })
  }
})();



