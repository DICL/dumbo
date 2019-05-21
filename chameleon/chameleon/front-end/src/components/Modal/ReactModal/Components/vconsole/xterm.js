import React, { Component, createRef } from 'react';

import { Terminal } from 'xterm';
import * as fit from 'xterm/lib/addons/fit/fit';
import * as attach from 'xterm/lib/addons/attach/attach';
import * as fullscreen from 'xterm/lib/addons/fullscreen/fullscreen';
import 'xterm/lib/addons/fullscreen/fullscreen.css';
import 'xterm/src/xterm.css';
import { Button } from 'semantic-ui-react'

class Xtrem extends Component {

  font_size = 12;
  rows = 30;
  columns = 105;
  width = 720;
  height= 408;

  constructor(props){
    super(props);
    this.appRef = createRef();

    this.state = {
         user_name: null,
         user_pass: null,
         is_login : false,
         is_open_console : false,
     };

    this.requestIDChange = this.requestIDChange.bind(this);
		this.requestPWChange = this.requestPWChange.bind(this);
    this.onSubmit = this.onSubmit.bind(this);
  }

  createToxterm(){
    const {font_size , rows , columns} = this

    Terminal.applyAddon(fit);      // Apply the `fit` addon
    Terminal.applyAddon(attach);  // Apply the `attach` addon
    Terminal.applyAddon(fullscreen);
    //let ws = this.ws;

    //let socket = new SockJS('/shell');
    this.xterm = new Terminal({
      "rows":rows,
      "cols":columns,
    });  // Instantiate the terminal
    // var ws = new WebSocket('ws://192.168.1.40:8084/echo');
    // this.xterm.attach(ws);

    this.xterm.setOption("fontSize",font_size);
    this.xterm.open(this.appRef.current);
    //this.xterm.toggleFullScreen();  // Now the terminal should occupy the full viewport

    //this.xterm.write('Hello from \x1B[1;3;31mxterm.js\x1B[0m $ ');

  }


  setFocus(){
    let xterm = this.xterm;
    if(xterm.focus){
      xterm.focus();
    }
  }

  action(type, data) {
    let action = Object.assign({
        type
    }, data);

    return JSON.stringify(action);
  }

  connectWebSocket(){
    let server_host = window.location.hostname;
    let {idx,  onRemove, host_name} = this.props;
    let {user_name,user_pass} = this.state;

    // webpack dev server 에서는 웹소켓이 동작하지 않음
    let server_port = (window.location.port === "4000") ? 8084 : window.location.port;
    let url = `ws://${server_host}:${server_port}/terminal?host=${host_name}`;
    //let url = 'ws://localhost:8084/terminal';

    this.ws = new WebSocket(url);
    let xterm = this.xterm;
    let ws = this.ws;


    this.setFocus();
    xterm.scrollLines(300);
    xterm.on('key',(e)=>{
      // console.log(e);
      ws.send(this.action("TERMINAL_COMMAND", {
            'command' : e
      }));
    })


    const {rows , columns , width ,height} = this

    ws.onopen = () => {
      //xterm.attach(ws);
      ws.send(this.action("TERMINAL_AUTH",{
        "user_name" : user_name, "user_pass" : user_pass
      }));
      ws.send(this.action("TERMINAL_INIT"));
      ws.send(this.action("TERMINAL_RESIZE" , {
        "columns": columns, "rows": rows,
        "width": width, "height": height,
      }));
    }

    ws.onerror = () => {

    }

    ws.onclose = () => {
      xterm.destroy();
      onRemove(idx);
    }

    ws.onmessage = (e) => {
      let data = JSON.parse(e.data);
      // console.log(data);

      // 소켓 통신을 json 으로 파싱하여 타입을 확인
      switch (data.type) {
        // 콘솔 내용 출력
        case "TERMINAL_PRINT":
            xterm.write(data.text);
          break;
        // 에러일경우
        case "TERMINAL_ALERT":
            //xterm.write(data.text);
            alert(data.text);
            this.closeXterm();
            onRemove(idx);
          break;
        default:

      }

      if(data.type === "TERMINAL_PRINT" ){

      }
    }



  }


  componentDidMount(){


  }

  componentWillUnmount(){
    this.closeXterm();
  }

  closeXterm(){
    if(this.ws){
      this.ws.close();
    }
    if(this.xterm){
      this.xterm.destroy();
      this.xterm = null;
    }
  }

  requestIDChange(event){
		this.setState({user_name: event.target.value});
	}
	requestPWChange(event){
		this.setState({user_pass: event.target.value});
	}

  onSubmit(){
    this.setState({is_login: true});
  }

  handleKeyPress = (e) => {
    if (e.charCode === 13) {
      e.onSubmit()
    }
  }

  componentDidUpdate(){
    const {is_login,is_open_console} = this.state;
    if(is_login && !is_open_console){
      this.createToxterm();
      this.connectWebSocket();
      this.setState({is_open_console: true});
    }
  }


  render () {
    const {is_login} = this.state;
    if(!is_login){
      return (
        <div className="login">
          <div className='ui form'>
            <div className='field'>
              <label>User Name</label>
              <input type="text" name="user_name" onChange={this.requestIDChange} placeholder='User Name'/>
            </div>
            <div className='field'>
              <label>Password</label>
              <input type="password" name="user_pass" onChange={this.requestPWChange} placeholder='Password'/>
            </div>
            <Button
              onClick={this.onSubmit}
              attached='bottom'
              onKeyPress={this.handleKeyPress}
              >
              Submit
            </Button>
          </div>
        </div>
      )
    }else{
      return (
        <div className="terminal">
          <div ref={this.appRef} onClick={(e)=>{ this.setFocus(); }}/>
        </div>
      )
    }

  }
}

export default Xtrem
