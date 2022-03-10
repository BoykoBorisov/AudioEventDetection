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
  const [test, setTest] = useState("Unset");
  const [data, setData] = useState();
  const [recordState, setRecordState] = useState(RecordState.STOP);
  const [displayType , setDisplayType] = useState("real");
  const intervalRef = useRef();
  const requestBufferRef = useRef([]);

  let parseResponse = (labelProbDict) => {
    return Object.entries(labelProbDict)
      .reduce(((arr, [key, value]) => [{title: key, value: value["probabilities"], id: value["id"]}, ...arr]),[])
      .filter(item => item["value"] > EPSILON);
  }

  let parseResponseAugmented = (labelProbDict) => {
    console.log("augmenting response");
    return Object.entries(labelProbDict)
      .reduce(((arr, [key, value]) => [{title: key, value: value["probabilities"], id: value["id"]}, ...arr]),[])
      // .filter(item => item["value"] > EPSILON)
      .map(item => {
        console.log(item);
        item["value"] /= coefs[item["id"]];
        return item;
      });
  }

  let handleTest = () => {
    setTest("Waiting for test response");
    inferPost((response) => {   
      setData(response);
    })
  }

  let updateChart = (response) => {
    let parsed;
    if (displayType === "real") parsed = parseResponse(response)
    else parsed = parseResponseAugmented(response);
    
    setData(parsed);
  }

  let prepareAndSendAudio = (data) => {
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
      <AudioReactRecorder state={recordState} onStop={prepareAndSendAudio}/>
      <div>
        <div>Display event probabilities based on:</div>
        <div onChange={handleRadioBtn}>
          <input type="radio" 
                 value="real" 
                 name="display_type" 
                 checked={displayType === "real"} /> Real values
          <input type="radio" 
                 value="augmented" 
                 name="display_type"
                 checked={displayType === "augmented"} /> Augmented values
        </div>
      </div>
      <button onClick={changeState}>Start/Stop</button>
      {data && <ChartRace data={data}/>}
    </div>
  );
}

export default App;
