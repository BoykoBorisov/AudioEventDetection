import axios from "axios";
import {SERVER_URL} from "./config";

let testGet = (callback) => {
  axios.get(SERVER_URL + "test", {})
    .then((response) => callback(response.data));
}

let inferPost = (formData) => {
  return axios.post(SERVER_URL + "infer", formData, {headers: {'Content-Type': 'multipart/form-data'}})
}

export {testGet, inferPost};