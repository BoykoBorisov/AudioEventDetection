import logo from './logo.svg';
import './App.css';
import {inferPost, testGet} from "./controller";
import React, { useRef, useState } from 'react';
import ChartRace from 'react-chart-race';
import AudioReactRecorder, {RecordState} from 'audio-react-recorder';
import coefs from './coefs.json'

import { EPSILON } from './config';

function App(props) {

  // console.log(coefs);
  const [data, setData] = useState();
  const [recordState, setRecordState] = useState(RecordState.STOP);
  const [displayType , setDisplayType] = useState("real");
  const intervalRef = useRef();
  const requestBufferRef = useRef([]);

  let parseResponse = (labelProbDict) => {
    return Object.entries(labelProbDict)
      .reduce(((arr, [key, value]) => [{title: key, value: value["probabilities"].toFixed(4), id: value["id"]}, ...arr]),[]);
  }

  let parseResponseAugmented = (labelProbDict) => {
    return Object.entries(labelProbDict)
      .reduce(((arr, [key, value]) => [{title: key, value: value["probabilities"], id: value["id"]}, ...arr]),[])
      .filter(entry => entry["value"] > coefs[entry["id"]]);
   
  }

  let updateChart = (response) => {
    let parsed;
    if (displayType === "real") parsed = parseResponse(response)
    else parsed = parseResponseAugmented(response);
    
    setData(parsed);
  }

  let prepareAndSendAudio = (data) => {
    console.log(data);
    let fd = new FormData();
    fd.append("1", data.blob, "1.wav");
    inferPost(fd).then((response) => {
      updateChart(response.data);
    })
  }

  let changeState = () => {
    if (recordState === RecordState.START) {
      clearInterval(intervalRef.current);
      setRecordState(RecordState.STOP);
      requestBufferRef.current = [];
    } else {
      setRecordState(RecordState.START);
      intervalRef.current = setInterval(() => {setRecordState(RecordState.STOP); setRecordState(RecordState.START)}, 10000);
    }
  }

  let handleRadioBtn = (event) => {
    setDisplayType(event.target.value);
  }
  return (
    <div className="App">
      <AudioReactRecorder state={recordState} onStop={prepareAndSendAudio} canvasHeight={250} type="wav"/>
      <div>
        <div>Display event probabilities based on:</div>
        <div onChange={handleRadioBtn}>
          <input type="radio" 
                 value="real" 
                 name="display_type" 
                 checked={displayType === "real"} /> All values
          <input type="radio" 
                 value="augmented" 
                 name="display_type"
                 checked={displayType === "augmented"} /> Values above threshold
        </div>
      </div>
      <button onClick={changeState}>Start/Stop</button>
      {data && <ChartRace data={data} itemHeight={27} />}
    </div>
  );
}

export default App;
