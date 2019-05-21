import React from 'react';
import _ from 'lodash';
import './default.css'
import { Grid } from 'semantic-ui-react'


//import * as modals_actions from '../../../../../actions/Modals';
import { connect } from 'react-redux';
import PlotlyJS from './PlotlyJS';

// idx : modal 고유값, onRemove : 해당 modal 삭제 메서드, data : host_name , metrics : 해당 호스팅 정보
const ReactModalSet = ({idx,  onRemove, data , node_list , create_modal }) => {
  console.log(data , node_list)
  //console.log(onRemove)
  // 해당 호스트 정보 추출
  let metrics = (node_list) ? _.filter(node_list,(item,index)=>{
    return _.indexOf(data,item.Hosts.host_name) !== -1;
  }) : [];

  let cpu_used , memory_used , disk_read_bytes , disk_write_bytes , network_bytes_in , network_bytes_out = [];

  let grape_cpu_used, grape_memory_used, grape_disk_read_write_bytes, grape_network_bytes_in_out = [];

  // 예외처리
  try {
    // cpu
    cpu_used = (metrics.length > 0) ?
                      _.chain(metrics)
                      .map((item)=>{return item.metrics.cpu.cpu_user; })
                      .value()
                    : 0
                    ;
  //  let cpu_avg_used = _.mean(cpu_used,(e)=>{return e;});

    // memory
    memory_used = (metrics) ?
                        _.chain(metrics)
                        .map((item)=>{ const { mem_total , mem_free } =  item.metrics.memory;  return ((mem_total - mem_free) / mem_total) * 100 ; })
                        .value()
                        : 0
                        ;
    //let memory_avg_used = _.mean(memory_used,(e)=>{return e;});


    // disk space
    // let disk_used = _.chain(metrics)
    //                   .map((item)=>{const { disk_free , disk_total } =  item.metrics.disk; return ((disk_total - disk_free) / disk_total) * 100; })
    //                   .value();
    // let disk_avg_used = _.mean(disk_used,(e)=>{return e;});


    // disk read
    disk_read_bytes = (metrics.length > 0) ?
                            _.chain(metrics)
                            .map((item)=>{return item.metrics.disk.read_bytes; })
                            .value()
                            : 0
                            ;

    // let disk_avg_read_bytes = _.mean(disk_read_bytes,(e)=>{return e;});


    // disk write
    disk_write_bytes = (metrics.length > 0) ?
                            _.chain(metrics)
                            .map((item)=>{return item.metrics.disk.write_bytes; })
                            .value()
                            : 0
                            ;

    // let disk_avg_write_bytes = _.mean(disk_write_bytes,(e)=>{return e;});



    network_bytes_in = (metrics.length > 0) ?
                            _.chain(metrics)
                            .map((item)=>{return item.metrics.network.bytes_in; })
                            .value()
                            : 0;

    // let network_avg_bytes_in = _.mean(network_bytes_in,(e)=>{return e;});


    network_bytes_out = (metrics.length > 0) ?
                            _.chain(metrics)
                            .map((item)=>{return item.metrics.network.bytes_out; })
                            .value()
                            : 0;

    // let network_avg_bytes_out = _.mean(network_bytes_out,(e)=>{return e;});
    // console.log(cpu_used , memory_used , disk_read_bytes , disk_write_bytes , network_bytes_in , network_bytes_out);

    grape_cpu_used = _.map(cpu_used,(item, index)=>{
      return {
        name : data[index] + ' USED ',
        value : item,
      }
      })
    grape_memory_used = _.map(memory_used,(item, index)=>{
      return {
        name : data[index] + ' USED ',
        value : item,
      }
      })
    grape_disk_read_write_bytes = _.reduce(
      _.map(data,(item, index)=>{
        return [{
          name : item + ' READ ',
          value : disk_read_bytes[index],
        },{
          name : item + ' WRITE ',
          value : disk_write_bytes[index],
        }]
      })
      ,
      (sum, n)=>{
        return sum.concat(n);
      }
    )
    grape_network_bytes_in_out = _.reduce(
      _.map(data,(item, index)=>{
        return [{
          name : item + ' IN ',
          value : network_bytes_in[index],
        },{
          name : item + ' OUT ',
          value : network_bytes_out[index],
        }]
      })
      ,
      (sum, n)=>{
        return sum.concat(n);
      }
    )

  } catch (e) {
    console.error(e);
  }



  let host_names = (data.length <= 1)? data[0] : data[0] +" "+ data.length + " hosts";


  //console.log(cpu_avg_used,memory_avg_used,disk_avg_used)

  return(
        <div className="DefaultModalItem ui card handle" style={{}}>
          <div className='content'>
            <div className='header cursor'>
                <span>{host_names}</span>
                {data.length <= 1 &&
                  <button
                    className="vconsole mini ui button"
                    onClick={ (e) => { e.stopPropagation(); /* onToggle 이 실행되지 않도록 함 */  create_modal( data ,'vconsole' );  }  }
                    >
                      vconsole
                  </button>
                }
                <button
                  className="popup-remove"
                  onClick={ (e) => { e.stopPropagation(); /* onToggle 이 실행되지 않도록 함 */  onRemove(idx);  }  }
                  >
                    &times;
                </button>
            </div>
          </div>
          {/* Modal 영역 */}
          <div className='content body'>
            <div className='description'>
              {data.length <= 1 &&
              <div>
                <h4>static system info</h4>
                <div>
                  Cluster Name : {metrics[0].Hosts.cluster_name}
                </div>
              </div>
              }

              <div className="default-graph-area">
                <h4>dynamic system info</h4>
                <Grid divided='vertically'>
                  <Grid.Row columns={2}>
                    <Grid.Column>
                      <PlotlyJS
                        title="CPU USED"
                        data={
                          grape_cpu_used
                          }
                        />
                    </Grid.Column>
                    <Grid.Column>
                      <PlotlyJS
                          title="MEMORY USED"
                          data={
                            grape_memory_used
                            }
                          />
                    </Grid.Column>
                  </Grid.Row>
                  <Grid.Row columns={2}>
                    <Grid.Column>
                      <PlotlyJS
                          title="DISK READ WRITE"
                          data={
                            grape_disk_read_write_bytes
                          }
                          chart_range={[]}
                        />
                    </Grid.Column>
                    <Grid.Column>
                      <PlotlyJS
                          title="NETWORK IN OUT"
                          data={
                            grape_network_bytes_in_out
                          }
                          chart_range={[]}
                        />
                    </Grid.Column>
                  </Grid.Row>
                </Grid>
                  {/*
                    <div>
                      CPU: {cpu_avg_used} % <br/>
                      MEM: {memory_avg_used} % <br/>
                      DISK: {disk_avg_used} %
                      NETWORK:
                    </div>
                    */}

              </div>
            </div>
          </div>

        </div>
  )
}

const mapStateToProps = (state) => {
    return {
        node_list: state.nodes.node_list,
    };
};

const mapDispatchToProps = (dispatch) => {
    //return bindActionCreators(actions, dispatch);
    return {

    };
};


export default connect(mapStateToProps, mapDispatchToProps) (ReactModalSet)
