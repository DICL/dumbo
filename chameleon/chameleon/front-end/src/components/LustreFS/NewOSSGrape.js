import React, { Component , createRef } from 'react';
import Plotly from 'plotly.js';
import _ from 'lodash';


class NewOSSgrape extends Component{

    constructor(props){
      super(props);
      this.appRef = createRef();
    }

    componentDidMount(){
      console.log(this.props);
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
      const { title,data } = this.props;

      let chart_data = data

      let chart_layout = {
        title : title,
        autosize: true,
    	  height: 200,
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

        yaxis: {
          // range: [0,100]
        },
      }

      let chart_option = {
        displayModeBar: false
      }

      this.chart = Plotly.plot( this.appRef.current , chart_data , chart_layout, chart_option );
    }

    updateChart(){
      const { data } = this.props;
      let chart_data = data
      //console.log(chart_data);

      let x = _.map(chart_data,(metric_data,metric_name)=>{ return [metric_data.x[ metric_data.x.length - 1 ]] });
      let y = _.map(chart_data,(metric_data,metric_name)=>{ return [ parseFloat(metric_data.y[ metric_data.y.length - 1 ]) ]  })

      let update = {
        x : x,
        y : y
      }
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
       Plotly.extendTraces(this.appRef.current, update, _.map(_.keys(chart_data),(data,index)=>{return index;}))
     } catch (e) {
       console.error(e);
     }

    }

    render() {
      return(
        <div className="column OSSGrape">
          <div ref={this.appRef}/>
        </div>
     );
    }

  }

export default NewOSSgrape;
