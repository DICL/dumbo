import React, { Component, createRef } from 'react';
import Plotly from 'plotly.js';
//import {randomColor} from '../../../../../services/utils';
// import Select from './Select';


// import _ from 'lodash';

class PlotlyJS extends Component{
  chart = null;

  constructor(props) {
    super(props);
    this.appRef = createRef();
    this.state = {
      SubHostsMenu : this.props.SubHostsMenu,
    }
  }

  // plotly js 그리기
  draw(){
    const { data } = this.props;
    this.chart_data = data;
    console.log(data);

    let olderTime = new Date(data.min);
    let futureTime = new Date(data.max);

    let chart_layout = {
      //title : application_id,
      autosize: true,
  	  height: 180,
  	  showlegend: true,

  	  margin: {
  	    l: 70,
  	    r: 40,
  	    b: 60,
  	    t: 40,
  	    pad: 10
  	  },

      xaxis: {
        autorange: true,
        rangeselector: {},
        type:'date',
        range: [olderTime , futureTime],
        // 181123 je.kim range view 끄기
        //rangeslider: {range: [futureTime.setMinutes(futureTime.getMinutes() - 2), futureTime]},
      },
      yaxis : {
        type : 'power',
        autorange : true
      },
    }

    if(this.props.title){
      chart_layout.title = this.props.title
    }
    if(this.props.y_label){
      // chart_layout.annotations= [{
      //   xref: 'paper',
      //   yref: 'paper',
      //   x: 0,
      //   xanchor: 'right',
      //   y: 1,
      //   yanchor: 'bottom',
      //   text: `${this.props.y_label}`,
      //   showarrow: false
      // }, {
      //   xref: 'paper',
      //   yref: 'paper',
      //   x: 1,
      //   xanchor: 'left',
      //   y: 0,
      //   yanchor: 'top',
      //   text: 'DateTime',
      //   showarrow: false
      // }]
      chart_layout.yaxis.title = this.props.y_label;
      chart_layout.yaxis.titlefont = {
        family: 'Courier New, monospace',
        size: 16,
        // color: '#7f7f7f'
      }
      chart_layout.xaxis.title = 'Time(sec)';
      chart_layout.xaxis.titlefont = {
        family: 'Courier New, monospace',
        size: 16,
        // color: '#7f7f7f'
      }
    }


    let chart_option = {
      displayModeBar: false
    }



    this.chart = Plotly.plot( this.appRef.current , data , chart_layout, chart_option );
  }

  // 11.26 je.kim 상단메뉴 제거


  // 컴포넌트가 생생될시 발생하는 이벤트 처리함수
  componentDidMount() {
    this.draw();

  }


  // 컴포넌트에 데이터가 수정될경우 이벤트 처리함수 및 노드 색상변경
  componentDidUpdate(){
  }




  componentWillUnmount() {
    // 컴포넌트가 사라질 때 인스턴스 제거
    if (this.chart !== null) {
      Plotly.purge(this.appRef.current);
      this.chart = null;
    }
  }

  render() {
    //console.log(this.props.data);
    //const { data,application_id,origenData } = this.props;

    return (
      <div>
        {/*
        <Select
          SubHostsMenu={this.state.SubHostsMenu}
          application_id={this.props.application_id}
          changeToggle={this.changeToggle}
          />
          */}
        <div className="PlotlyJS" ref={this.appRef}/>
      </div>

    );
  }
}

export default PlotlyJS;
