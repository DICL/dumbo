  import React, { Component, createRef } from 'react';
  import Plotly from 'plotly.js';
  import './YarnMonitoring.css'
    class MemoryStatus extends Component{

      constructor(props){
        super(props);
        this.appRef1 = createRef();
      }

      componentDidMount(){
        this.draw();
      }

      componentWillUnmount(){
        // 컴포넌트가 사라질 때 인스턴스 제거
        if (this.chart !== null) {
          try {
            Plotly.purge(this.appRef1.current);
            this.chart = null;
          } catch (e) {
            console.error(e);
          }
        }
      }

      componentDidUpdate(){
        this.updateChart();
      }

      // 최초로 컴퍼넌트가 시작될때에 그래프를 그리는 매서드
      draw(){
        const {yarn_status} = this.props;
        const time = new Date();
        let olderTime = time.setMinutes(time.getMinutes() - 1);
        let futureTime = time.setMinutes(time.getMinutes() + 1);

        let {totalMB,availableMB,allocatedMB,reservedMB} = yarn_status;

        var data = [
          {
            x: [time],
            y: [totalMB],
            mode: 'lines',
            name: 'totalMB',
            // line: {color: '#80CAF6'}
          },{
            x: [time],
            y: [availableMB],
            mode: 'lines',
            name: 'availableMB',
            // line: {color: '#80CAF6'}
          },{
            x: [time],
            y: [allocatedMB],
            mode: 'lines',
            name: 'allocatedMB',
            // line: {color: '#80CAF6'}
          },{
            x: [time],
            y: [reservedMB],
            mode: 'lines',
            name: 'reservedMB',
            // line: {color: '#80CAF6'}
          }
        ]
        let layout = {
          title: 'Memory Status',
          autosize: true,
          height: 200,
          showlegend:true,
          // legend: {
          //   "orientation": "h",
          //   "xanchor":"center",
          //   "yanchor":"middle",
          //   "y":-0.5, // play with it
          //   "x":0.5   // play with it
          // },
          margin: {
      	    l: 40,
      	    r: 40,
      	    b: 40,
      	    t: 40,
      	    pad: 10
      	  },
          yaxis: {
            // range: [0 , 100],
            // range: [0]
          },
          xaxis: {
            autorange: true,
            rangeselector: {},
            type:'date',
            range: [olderTime , futureTime],
          },
        };

        let chart_option = {
          displayModeBar: false
        }

        this.chart = Plotly.plot( this.appRef1.current , data , layout, chart_option);
      }

      // 차트 라인 업데이트
      updateChart(){
        // console.log(this.props.yarn_status);
        const {yarn_status} = this.props;
        let {totalMB,availableMB,allocatedMB,reservedMB} = yarn_status;
        const time = new Date();
        let update = {
          x : [
            [time],
            [time],
            [time],
            [time],
          ],
          y : [
            [totalMB],
            [availableMB],
            [allocatedMB],
            [reservedMB],
          ]
        }
        var olderTime = time.setMinutes(time.getMinutes() - 1);
        var futureTime = time.setMinutes(time.getMinutes() + 1);
        var minuteView = {
         xaxis: {
           type: 'date',
           range: [olderTime,futureTime]
         }
       };


       try {
         Plotly.relayout(this.appRef1.current, minuteView);
         Plotly.extendTraces(this.appRef1.current, update, [0,1,2,3])
       } catch (e) {
         console.error(e);
       }

      }


      render() {
        // const {yarn_status} = this.props;
        return(
          <div id="YarnMonitoring" ref={this.appRef1}></div>
       );
      }

    }

    export default MemoryStatus;
