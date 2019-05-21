import React, { Component, createRef } from 'react';
import Plotly from 'plotly.js';


class PlotlyJS extends Component{
  chart = null;

  randomColor(){
    var letters = '0123456789ABCDEF';
    var color = '#';
    for (var i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  }

  constructor(props) {
    super(props);
    this.appRef = createRef();
  }

  // this.props 기본값
  static defaultProps = {
    chart_range : [0,100],
    height : 250
  }

  // plotly js 그리기
  draw(){

    const { data , chart_range , title , height } = this.props;

    var time = new Date();

    var chart_layout = {
      title : title,
      autosize: true,
  	  height: height,
  	  showlegend: true,
      legend: {
						  "orientation": "h",
						  xanchor:"center",
						  yanchor:"middle",
						  y:-1, // play with it
						  x:0.5   // play with it
			},
  	  margin: {
  	    l: 40,
  	    r: 40,
  	    b: 40,
  	    t: 40,
  	    pad: 10
  	  },
  	  yaxis: {
        range: chart_range
      },
    }

    var chart_option = {
      displayModeBar: false
    }


    try {
      var set_data = data.map(({name,value}) => {
        return {
          x: [time],
          y: [value],
          mode: 'lines',
          type: 'scatter',
          name: name,
          line: {color: this.randomColor()}
        };
      });

      this.chart = Plotly.plot( this.appRef.current , set_data , chart_layout, chart_option );
    } catch (e) {
      console.error(e);
    }

  }

  updateChart(){
    const { data } = this.props;
    var time = new Date();
    var olderTime = time.setMinutes(time.getMinutes() - 1);
    var futureTime = time.setMinutes(time.getMinutes() + 1);

    // var update = {
    //   x:  [[time]],
    //   y: [[this.props.data]]
    // }

    try {
      var update = {
        x:  data.map(({name,value}) => {
          return [time];
        }),
        y: data.map(({name,value}) => {
          return [value];
        })
      }
      var minuteView = {
          xaxis: {
            type: 'date',
            range: [olderTime,futureTime]
          }
        };
        //console.log(data.map((key, index)=>{return index;}));
        Plotly.relayout(this.appRef.current, minuteView);
        Plotly.extendTraces(this.appRef.current, update, data.map((key, index)=>{return index;}))
    } catch (e) {
      console.error(e);
    }

    // Plotly.relayout(this.appRef.current, minuteView);
    // Plotly.extendTraces(this.appRef.current, update, data.map((key, index)=>{return index;}))
    // console.log(update,data.map((key, index)=>{return index;}));


  }

  // 컴포넌트가 생생될시 발생하는 이벤트 처리함수
  componentDidMount() {
    const { data } = this.props;
    if (typeof data !== 'undefined' && data !== null) {
      this.draw();
    }
  }

  // 컴포넌트에 데이터가 수정될경우 이벤트 처리함수 및 노드 색상변경
  componentDidUpdate(){
    const { data } = this.props;
    if (typeof data !== 'undefined' && data !== null) {
      this.updateChart();
    }
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
    const { data } = this.props;
    if (typeof data !== 'undefined' && data !== null) {
      return (
        <div className="PlotlyJS" ref={this.appRef}/>
      );
    }else{
      return(<div></div>)
    }

  }
}

export default PlotlyJS;
