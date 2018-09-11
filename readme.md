# Write Me a Sci-Fi

The only thing better than a terrible Machine Learning project is a terrible machine learning project in a web app.

This project uses the model trained as part of my [Sci-fi-nn project](https://github.com/Ibexian/scifi-nn/) and makes it interactive [on the web](https://william.kamovit.ch/writeMeAScifi/). The app slowly (and I do mean slowly) generates text based on user input.

Since Parcel alllows direct integration with Rust => Web Assembly the sampling function is written in Rust.

### To Serve Locally
`parcel index.html`


### Tools:
- Keras-js (technically ibexian-keras-js since keras-js is broken and no longer supported)
- Parcel
- numjs
- Bulma Css
- Babel
- Rust