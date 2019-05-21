import React, { Component, createRef } from 'react';
import Plotly from 'plotly.js';
import './YarnMonitoring.css'
  class vCoreStatus extends Component{

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

      let {reservedVirtualCores,availableVirtualCores,allocatedVirtualCores} = yarn_status;

      var data = [
        {
          x: [time],
          y: [reservedVirtualCores],
          mode: 'lines',
          name: 'resrvedVirtualCores',
          // line: {color: '#80CAF6'}
        },{
          x: [time],
          y: [availableVirtualCores],
          mode: 'lines',
          name: 'availableVirtualCores',
          // line: {color: '#80CAF6'}
        },{
          x: [time],
          y: [allocatedVirtualCores],
          mode: 'lines',
          name: 'allocatedVirtualCores',
          // line: {color: '#80CAF6'}
        }
      ]
      let layout = {
        title: 'vCore Status',
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
      let {reservedVirtualCores,availableVirtualCores,allocatedVirtualCores} = yarn_status;
      const time = new Date();
      let update = {
        x : [
          [time],
          [time],
          [time],
        ],
        y : [
          [reservedVirtualCores],
          [availableVirtualCores],
          [allocatedVirtualCores],
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
       Plotly.extendTraces(this.appRef1.current, update, [0,1,2])
     } catch (e) {
       console.error(e);
     }

    }


    render() {
      // const {yarn_status} = this.props;
      return(
        <div id="YarnMonitoring" ref={this.appRef1}>
          {/*
            <table className="yarn-status-table">
              <tbody>
                <tr>
                  <td>totalVirtualCores</td>
                  <td>{yarn_status.totalVirtualCores}</td>
                  <td>totalMB</td>
                  <td>{yarn_status.totalMB}</td>
                  <td>totalNodes</td>
                  <td>{yarn_status.totalNodes}</td>
                  <td>appsCompleted</td>
                  <td>{yarn_status.appsCompleted}</td>
                    <td>containersAllocated</td>
                    <td>{yarn_status.containersAllocated}</td>
                </tr>
                <tr>
                  <td>allocatedVirtualCores</td>
                  <td>{yarn_status.allocatedVirtualCores}</td>
                  <td>availableMB</td>
                  <td>{yarn_status.availableMB}</td>
                  <td>activeNodes</td>
                  <td>{yarn_status.activeNodes}</td>
                  <td>appsFailed</td>
                  <td>{yarn_status.appsFailed}</td>

                  <td>containersPending</td>
                  <td>{yarn_status.containersPending}</td>
                </tr>
                <tr>
                  <td>availableVirtualCores</td>
                  <td>{yarn_status.availableVirtualCores}</td>
                  <td>reservedMB</td>
                  <td>{yarn_status.reservedMB}</td>
                  <td>unhealthyNodes</td>
                  <td>{yarn_status.unhealthyNodes}</td>
                  <td>appsKilled</td>
                  <td>{yarn_status.appsKilled}</td>
                  <td>containersReserved</td>
                  <td>{yarn_status.containersReserved}</td>
                </tr>
                <tr>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td>decommissionedNodes</td>
                  <td>{yarn_status.decommissionedNodes}</td>
                  <td>appsPending</td>
                  <td>{yarn_status.appsPending}</td>
                </tr>
                <tr>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td>lostNodes</td>
                  <td>{yarn_status.lostNodes}</td>
                  <td>appsRunning</td>
                  <td>{yarn_status.appsRunning}</td>
                </tr>
                <tr>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td>appsSubmitted</td>
                  <td>{yarn_status.appsSubmitted}</td>
                </tr>
              </tbody>
            </table>
          */}
        </div>
     );
    }

  }

  export default vCoreStatus;
