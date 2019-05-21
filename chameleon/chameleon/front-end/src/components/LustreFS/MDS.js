import React, { Component , createRef} from 'react';
import Plotly from 'plotly.js';
// import {randomColor,timeConverter} from '../../services/utils';
import _ from 'lodash';

class MDS extends Component{

  constructor(props){
    super(props);
    this.appRef = createRef();
  }

  componentDidMount(){
    // console.log(this.props.mds_metric);
    this.draw();
  }

  componentWillUnmount(){
    // 컴포넌트가 사라질 때 인스턴스 제거
    if (this.chart !== null) {
      Plotly.purge(this.appRef.current);
      this.chart = null;
    }
  }

  componentDidUpdate(){
    this.updateChart();
  }

  draw(){
    const { mds_metric } = this.props;
    //console.log(mds_metric);

    // const time = new Date();
    // let olderTime = time.setMinutes(time.getMinutes() - 1);
    // let futureTime = time.setMinutes(time.getMinutes() + 0);
    // let range_olderTime = time.setMinutes(time.getMinutes() - 10);
    // let range_futureTime = time.setMinutes(time.getMinutes() + 0);

    // console.log(range_olderTime,range_futureTime);

    let chart_layout = {
      //title : application_id,
      autosize: true,
  	  height: 300,
  	  showlegend: true,

  	  margin: {
  	    l: 40,
  	    r: 40,
  	    b: 40,
  	    t: 40,
  	    pad: 10
  	  },

      xaxis: {
        autorange: true,
        rangeselector: {},
        type:'date',
        // rangeslider: {},
      },
    }

    let chart_option = {
      displayModeBar: true
    }


    let set_data = _.map(mds_metric,(metric_data,metric_name)=>{
      return {
        x: _.map(metric_data,(data)=>{ return (data[1] ) }),
        y: _.map(metric_data,(data)=>{ return data[0] }),
        mode: 'lines',
        type: 'scatter',
        name: `${metric_name}`,
        //line: {color: randomColor()},
      }
    })

    //console.log(set_data);

    this.chart = Plotly.plot( this.appRef.current , set_data , chart_layout, chart_option );

  }

  // 차트 라인 업데이트
  updateChart(){
    const { mds_metric } = this.props;

    const time = new Date();

    let x = _.map(mds_metric,(metric_data,metric_name)=>{ return [metric_data[ metric_data.length - 1 ][1]] });
    let y = _.map(mds_metric,(metric_data,metric_name)=>{ return [ parseFloat(metric_data[ metric_data.length - 1 ][0]) ]  })

    let update = {
      x : x,
      y : y
    }
    //console.log(update,mds_metric);

    var olderTime = new Date(_.min(x));
    var futureTime = new Date(_.max(x));
    var minuteView = {
     xaxis: {
       type: 'date',
       range: [olderTime,futureTime]
     }
   };


   try {
     Plotly.relayout(this.appRef.current, minuteView);
     Plotly.extendTraces(this.appRef.current, update, _.map(_.keys(mds_metric),(data,index)=>{return index;}))
   } catch (e) {
     console.error(e);
   }
  }


  render() {
    return(
      <div ref={this.appRef}/>
   );
  }

}

export default MDS;
