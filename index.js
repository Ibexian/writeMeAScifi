import KerasJS from 'ibexian-keras-js';
import ndarray from 'ndarray';
import unpack from 'ndarray-unpack';
import nj from 'numjs';
import sampling from 'discrete-sampling';
import charJson from './char.json';
import 'babel-polyfill';
const modelBin = require('./final_model.bin');

const charToId = charJson.charToId;
const reverseDictionary = charJson.idToChar;
const TOTALCHARS = Object.keys(charToId).length;
const INPUTSIZE = 50;
var targetSize = 140;
var outputText = '';

const updateProgress = (percent) => {
    const progressBar = document.querySelector('progress');
    progressBar.value = percent;
    if (percent === 100) {
        document.querySelector('#progress').classList.add('is-invisible');
        document.querySelector('#inputs').classList.remove('is-invisible');
    }
};

const captureSeedText = () => {
    //Grab any input text from the page
    const textInput = document.querySelector('#seedText');
    const seed = textInput.value.trim();
    //convert chars to an id array
    let seedArr = seed.split('').map((char) => Number.parseInt(charToId[char] || 0));
    //seed an array with random numbers
    let randArr = createInitArray(INPUTSIZE, TOTALCHARS);
    //Combine the id and the random number arrays up to INPUTSIZE
    if (seedArr.length > INPUTSIZE) {
        seedArr = seedArr.slice(-INPUTSIZE);
    } else {
        randArr = randArr.slice(0, (randArr.length - seedArr.length))
        seedArr = [...randArr, ...seedArr];
    }
    return [seed, seedArr];
};

const sample = (arr, sampleRate=1) => {
    var preds = nj.array(arr);
    preds = nj.log(preds);
    preds = nj.divide(preds, sampleRate);
    let exp_preds = nj.exp(preds);
    //normalize inputs
    preds = nj.divide(exp_preds, nj.sum(exp_preds));
    //Create an array of probabilities
    let probabilities = sampling.Multinomial(1, preds.tolist(), 1);
    //Return the one selected id based on probs
    return probabilities.draw().reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
};

const createInitArray = (length, max) => {
    //Create an array of random ints of 'length' size between 0 and 'max'
    return nj.round(nj.multiply(nj.random([length]), (max - 1))).tolist();
};

const enableSubmit = () => {
    const submitButton = document.querySelector('#submitButton');
    const textInput = document.querySelector('#seedText');
    //(un)bind submit event
    submitButton.removeEventListener('click', submitPrediction);
    submitButton.addEventListener('click', submitPrediction);
    textInput.removeEventListener("keypress", submitOnEnter);
    textInput.addEventListener("keypress", submitOnEnter);
    //Enable Button
    submitButton.classList.remove('is-loading');
    submitButton.removeAttribute('disabled');
};

const diableSubmit = () => {
    const submitButton = document.querySelector('#submitButton');
    const textInput = document.querySelector('#seedText');
    //unbind submit event
    textInput.removeEventListener("keypress", submitOnEnter);
    //Disable Button
    submitButton.classList.add('is-loading');
    submitButton.setAttribute('disabled', true);
};

const displayResult = () => {
    const resultTextBox = document.querySelector('.result-box textarea');
    //Print output to box
    resultTextBox.value = outputText;
    //clear text input
    document.querySelector('#seedText').value = '';
}

const submitOnEnter = (e) => {
    if (e.which === 13) {
        submitPrediction();
    }
};

const addTextAndCursor = (newChar) => {
    const resultBox = document.querySelector('.result-box textarea');
    //Remove Cursor char
    var results = resultBox.value.split('').slice(0, -1).join('');
    //Add new char with cursor at end
    resultBox.value = results += newChar += `\u2759`;
};

const submitPrediction = () => {
    diableSubmit();
    let [seed, seedArr] = captureSeedText();
    outputText = seed += `\u2759`;
    predictText('', seedArr);
    displayResult();
};
const predictText = (predictionResult, seedArr) => {
    const inputData = {
        input: new Float32Array(seedArr)
    };
    var t0 = performance.now();
    model.predict(inputData)
    .then(outputData => {
        var output = new Float32Array(outputData.output);
        var predictions = new ndarray(output, [INPUTSIZE, TOTALCHARS]);
        var nextCharPrediction = unpack(predictions)[INPUTSIZE - 1];
        var ix = sample(nextCharPrediction, 0.5);
        var predictedChar = reverseDictionary[ix.toString()];
        predictionResult += predictedChar;
        addTextAndCursor(predictedChar)
        if (predictionResult.length < targetSize) {
            seedArr.shift()
            seedArr.push(ix);
            var t1 = performance.now();
            console.log(t1 - t0);
            predictText(predictionResult, seedArr);
        } else {
            outputText += predictionResult;
            enableSubmit();
        }
    })
    .catch(err => {
        console.error(err);
        enableSubmit();
        // handle error
    })
};

const model = new KerasJS.Model({
    filepath: modelBin,
    pauseAfterLayerCalls: true
});

model.events.on('loadingProgress', updateProgress);

model.ready().then(() => {
    document.querySelector('.result-box textarea').value = '';
    enableSubmit();
    //TODO Style
    //TODO Animations
    //TODO Speed up Loop? - Each prediction isn't too bad - but doing 140 predictions is awful
    //Try without the sampling or seedarray mods - maybe using timing
});

