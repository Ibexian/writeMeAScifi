import KerasJS from 'ibexian-keras-js';
import ndarray from 'ndarray';
import unpack from 'ndarray-unpack';
import nj from 'numjs';
import charJson from './char.json';
import rustWasm from './rust/cargo.toml';
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

const predictionToChar = async(output, predictionResult, seedArr) => {
    var output = new Float32Array(output.output);
    var predictions = new ndarray(output, [INPUTSIZE, TOTALCHARS]);
    var nextCharPrediction = unpack(predictions)[INPUTSIZE - 1];
    var ix = await await rustWasm.sample(0.5, nextCharPrediction).result;
    var predictedChar = reverseDictionary[ix.toString()];
    // console.log("prediction", predictedChar, ix)
    predictionResult += predictedChar;
    addTextAndCursor(predictedChar)
    if (predictionResult.length < targetSize) {
        seedArr.shift()
        seedArr.push(ix);
        predictText(predictionResult, seedArr);
    } else {
        outputText += predictionResult;
        enableSubmit();
    }
};
const predictText = async (predictionResult, seedArr) => {
    const inputData = {
        input: new Float32Array(seedArr)
    };
    model.predict(inputData)
    .then(outputData => {
        predictionToChar(outputData, predictionResult, seedArr);
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
});

