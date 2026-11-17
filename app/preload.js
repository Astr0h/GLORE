const { contextBridge, ipcRenderer, webUtils } = require('electron');

const portArg = process.argv.find((a) => a.startsWith('--glore-port='));
let port = portArg ? Number(portArg.split('=')[1]) : 8765;

contextBridge.exposeInMainWorld('glore', {
  engineUrl: () => `http://127.0.0.1:${port}`,
  // Electron 32+ removed File.path; this is the supported way to get a dropped file's path
  pathForFile: (file) => webUtils.getPathForFile(file),
  pickFiles: () => ipcRenderer.invoke('pick-files'),
  pickCsv: () => ipcRenderer.invoke('pick-csv'),
  openLog: () => ipcRenderer.invoke('open-log'),
  restartEngine: async () => { port = await ipcRenderer.invoke('restart-engine'); return port; },
  onEngineExit: (cb) => ipcRenderer.on('engine-exit', (_e, code) => cb(code)),
});
