import logo from './logo.svg';
import './App.css';
import {inferPost, testGet} from "./controller";
import React, { useRef, useState } from 'react';
import ChartRace from 'react-chart-race';
import AudioReactRecorder, {RecordState} from 'audio-react-recorder';

import { EPSILON } from './config';

function App(props) {

  const [test, setTest] = useState("Unset");
  const [data, setData] = useState();
  const [recordState, setRecordState] = useState(RecordState.STOP);
  const intervalRef = useRef();
  const requestBufferRef = useRef([]);

  let parseResponse = (labelProbDict) => {
    return Object.entries(labelProbDict)
      .reduce(((arr, [key, value]) => [{title: key, value: value["probabilities"], id: value["id"]}, ...arr]),[])
      .filter(item => item["value"] > EPSILON);
  }

  let handleTest = () => {
    setTest("Waiting for test response");
    inferPost((response) => {
      let parsed = parseResponse(response);
      console.log(parsed);
      setData(parsed);
    })
  }

  let updateChart = (response) => {
    setData(parseResponse(response));
  }

  let prepareAndSendAudio = (data) => {
    // while (requestBufferRef.current.length > 3) {
    //   requestBufferRef.current.shift();
    // }
    // requestBufferRef.current.push(data.blob);
    console.log(new Date());
    // if (requestBufferRef.current.length === 4) {
    let fd = new FormData();
    //   for (let i = 0; i < requestBufferRef.current.length; i++)
    //     fd.append(i, requestBufferRef.current[i], i + ".wav");
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
  return (
    <div className="App">
      <AudioReactRecorder state={recordState} onStop={prepareAndSendAudio}/>
      <button onClick={changeState}>Start/Stop</button>
      <button onClick={handleTest}>Test Btn</button><p>{test}</p>
      {data && <ChartRace data={data}/>}
    </div>
  );
}

export default App;
