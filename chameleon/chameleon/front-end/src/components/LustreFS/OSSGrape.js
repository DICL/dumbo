import React, { Component , createRef } from 'react';
import Plotly from 'plotly.js';
  class OSSGrape extends Component{

    constructor(props){
      super(props);
      this.appRef = createRef();
    }

    componentDidMount(){
      console.log(this.props.data);
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

    }

    draw(){
      const { data } = this.props;

      let chart_data = [
        data.set_data
      ]

      let chart_layout = {
        title : data.title,
        autosize: true,
    	  height: 200,
    	  showlegend: false,

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


    render() {
      return(
        <div className="column OSSGrape">
          <div ref={this.appRef}/>
        </div>

     );
    }

  }

  export default OSSGrape;
