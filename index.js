import KerasJS from 'ibexian-keras-js';
import ndarray from 'ndarray';
import unpack from 'ndarray-unpack';
import nj from 'numjs';
import sampling from 'discrete-sampling';
import charJson from './char.json';
import 'babel-polyfill';

const charToId = charJson.charToId;
const reverseDictionary = charJson.idToChar;
const TOTALCHARS = Object.keys(charToId).length;

const updateProgress = (percent) => {
    const progressBar = document.querySelector('progress');
    progressBar.value = percent;
    if (percent === 100) {
        document.querySelector('#progress').classList.add('is-invisible');
        document.querySelector('#inputs').classList.remove('is-invisible');
    }
}

const sample = (arr, sampleRate=1) => {
    var preds = nj.array(arr);
    preds = nj.log(preds);
    preds = nj.divide(preds, sampleRate);
    let exp_preds = nj.exp(preds);
    //normalize inputs
    preds = nj.divide(exp_preds, nj.sum(exp_preds));
    //Create an array of probabilities
    let probabilities = sampling.Multinomial(1, preds.tolist(), 1);
    return probabilities.draw().reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
}

const model = new KerasJS.Model({
    filepath: 'final_model.bin',
    pauseAfterLayerCalls: true

});

model.events.on('loadingProgress', updateProgress)

model.ready().then(() => {
    //TODO Random array init
    //TODO Add seed
    // input data object keyed by names of the input layers
    // or `input` for Sequential models
    // values are the flattened Float32Array data
    // (input tensor shapes are specified in the model config)
    const inputData = {
        input: new Float32Array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )};

    // make predictions
    return model.predict(inputData)
})
.then(outputData => {
    var output = new Float32Array(outputData.output);
    var predictions = new ndarray(output, [50, TOTALCHARS]);
    var nextCharPrediction = unpack(predictions)[49];
    var ix = sample(nextCharPrediction, 0.25);
    console.log(ix.toString)
    console.log(reverseDictionary[ix.toString()])
    //console.log(predictions)
    // outputData is an object keyed by names of the output layers
    // or `output` for Sequential models
    // e.g.,
    // outputData['fc1000']
})
.catch(err => {
    console.error(err)
    // handle error
})
