import React, { Component, createRef } from 'react';
import Plotly from 'plotly.js';
import _ from 'lodash';
import './AppMonitor.css';
import Selected from './Select';
import {randomColor/*,timeConverter*/} from '../../../../services/utils';




class PlotlyJS extends Component{

  constructor() {
      super();
      this.appRef = createRef();
      this.state = {}
      this.containers = {}
  }

  viewOrHideTraces = (e,container_id)=>{
    //console.log(e.checked,container_id);
    // 체크박스 체크
    let tmp_containers = this.containers;
    let index = tmp_containers[container_id]['index'];

    if (e.checked) {
      Plotly.restyle(this.appRef.current, 'visible', true, (2*index));
      Plotly.restyle(this.appRef.current, 'visible', true, (2*index)+1);
      this.containers[container_id]['isShow'] = true;
    } else {
      Plotly.restyle(this.appRef.current, 'visible', 'legendonly', (2*index));
      Plotly.restyle(this.appRef.current, 'visible', 'legendonly', (2*index)+1);
      this.containers[container_id]['isShow'] = false;
    }
  }

  draw(){
    const time = new Date();
    let olderTime = time.setMinutes(time.getMinutes() - 1);
    let futureTime = time.setMinutes(time.getMinutes() + 1);

    let range_olderTime = time.setMinutes(time.getMinutes() - 10);
    let range_futureTime = time.setMinutes(time.getMinutes() + 0);

    let layout = {
      // title: 'process monitoring',
      autosize: true,
      showlegend:true,
      margin: {
  	    l: 40,
  	    r: 40,
  	    b: 40,
  	    t: 40,
  	    pad: 10
  	  },
      yaxis: {
        range: [0 , 100]
        // range: [0]
      },
      xaxis: {
        autorange: true,
        rangeselector: {},
        type:'date',
        range: [olderTime , futureTime],
        rangeslider: {range: [range_olderTime, range_futureTime]},
      },
    };
    this.chart = Plotly.plot( this.appRef.current , [] , layout);
  }

  // 차트 라인 저장
  updateStorage(){
    const limit_conut = 300; // 그래프 Traces가 남아있는 시간
    const {data} = this.props;
    var time = new Date();
    let tmp_containers = this.containers;

    // 만약 더이상 데이터가 없을경우 삭제
    _.each(tmp_containers,(item,index)=>{
      if(item['kill_count'] > limit_conut){
        let deleteIndex = item['index'];
        try {
          Plotly.deleteTraces(this.appRef.current, [(2*deleteIndex),((2*deleteIndex)+1)] )
        } catch (e) {
          console.error(e);
        }
        delete tmp_containers[index];

        var i = 0;
        _.each(tmp_containers,(item,index)=>{
          tmp_containers[index]['index'] = i;
          ++ i;
        });

      }else{
        tmp_containers[index]['kill_count'] ++ ;
      }
    });



    // 받은 데이터들을 컨테이너 아이디에 맞추어서 정렬
    _.each(data,(item,index)=>{
      // 기존의 데이터일경우
      if(tmp_containers[item.container_id]){
        tmp_containers[item.container_id]['kill_count'] = 0;
        tmp_containers[item.container_id]['list'].push(item);
      }
      // 새로운 범례가 추가될 경우
      else{
        let index = Object.keys(tmp_containers).length;
        tmp_containers[item.container_id] = {};
        tmp_containers[item.container_id]['kill_count'] = 0;
        tmp_containers[item.container_id]['list'] = [];
        tmp_containers[item.container_id]['list'].push(item);
        tmp_containers[item.container_id]['index'] = index;
        tmp_containers[item.container_id]['pid'] = item.pid;
        tmp_containers[item.container_id]['application_id'] = item.application_id;
        tmp_containers[item.container_id]['container_id'] = item.container_id;
        tmp_containers[item.container_id]['node'] = item.node;
        tmp_containers[item.container_id]['checked'] = true;

        // 라인추가 (cpu)
        //let tmpTime = new Date(item.create_date);
        Plotly.addTraces(this.appRef.current,{
            x :[time],
            y : [item.cpu_used],
            name: `${item.pid} CPU`,
            marker: {color: randomColor()},
            mode: 'lines',
            type: 'scatter',
        }, (2*index) );
        // 라인추가(mem)
        Plotly.addTraces(this.appRef.current,{
            x :[time],
            y: [item.mem_used],
            name: `${item.pid} MEM`,
            marker: {color: randomColor()},
            mode: 'lines',
            type: 'scatter',
        }, (2*index)+1);
      }

    });


    // 기존의 저장되어 있는 정보를 교체
    this.containers = tmp_containers;
    // console.log(tmp_containers,data);

    // 그래프에 점추가
    var update = {
      x : [],
      y : []
    }
    _.each(data,(item,index)=>{
      update.x.push([time]);
      update.y.push([item.cpu_used]);
      update.x.push([time]);
      update.y.push([item.mem_used]);
    });

    var tempTracesIndex = [];
    _.each(data,(item,index)=>{
      let tempIndex = tmp_containers[item.container_id]['index'];
      tempTracesIndex.push((2*tempIndex));
      tempTracesIndex.push((2*tempIndex)+1);
    });

    // console.log(update, tempTracesIndex);

    try {
      Plotly.extendTraces(this.appRef.current, update, tempTracesIndex )
    } catch (e) {
      console.error(e);
    }

  }

  updateChart(){
    const time = new Date();
    let olderTime = time.setMinutes(time.getMinutes() - 1);
    let futureTime = time.setMinutes(time.getMinutes() + 1);
    let range_olderTime = time.setMinutes(time.getMinutes() - 10);
    let range_futureTime = time.setMinutes(time.getMinutes() + 0);

    let minuteView = {
        xaxis: {
          type: 'date',
          rangeselector: {},
          range: [olderTime,futureTime],
          rangeslider: {range: [range_olderTime, range_futureTime]},
        }
    };
    Plotly.relayout(this.appRef.current, minuteView);
  }





  // 컴포넌트가 생생될시 발생하는 이벤트 처리함수
  componentDidMount() {
     this.draw();
  }

  // 컴포넌트에 데이터가 수정될경우 이벤트 처리함수 및 노드 색상변경
  componentDidUpdate(){
    this.updateStorage();
    this.updateChart();
  }

  componentWillUnmount() {
    // 컴포넌트가 사라질 때 인스턴스 제거
    if (this.chart !== null) {
      Plotly.purge(this.appRef.current);
      this.chart = null;
    }
    this.containers = null;
  }

  render() {
    // console.log(this.props.data);
    return(
      <div style={{width:'100%'}}>
        <div>
          <Selected
            plotly={this.appRef}
            data={this.containers}
            viewOrHideTraces={this.viewOrHideTraces}
            />
        </div>
        <div>
          <div className="PlotlyJS" ref={this.appRef}/>
        </div>
      </div>
    )
  }

}

export default PlotlyJS;
