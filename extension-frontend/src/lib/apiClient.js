import axios from "axios";

const NODE_BASE =
  import.meta.env.VITE_NODE_API || "http://localhost:5000";

const PYTHON_BASE =
  import.meta.env.VITE_PYTHON_API || "http://localhost:8001";


export const nodeApi = axios.create({
  baseURL: NODE_BASE,
});

export const pythonApi = axios.create({
  baseURL: PYTHON_BASE,
  timeout: 120000,
});