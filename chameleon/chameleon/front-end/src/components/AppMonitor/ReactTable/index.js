import React, { Component } from 'react';
import ReactTable from 'react-table'
import 'react-table/react-table.css'
import { Button } from 'semantic-ui-react'
import './AppHistory.css'
import * as utils from '../../../services/utils';



class AppHistory extends Component{

  constructor() {
      super();
      this.state = {
        resize: false
      }
      this.child = React.createRef();
    }


  render() {
    //console.log(this.props.yarn_job_list);
    const data = this.props.yarn_job_list
    const data_count = data.length
    const columns = [
      {
        Header: 'ID',
        accessor: 'id',
        maxWidth: 100,
      },
      {
        Header: 'User',
        accessor: 'user',
        maxWidth: 80,
      },
      {
        Header: 'Application Type',
        accessor: 'applicationType',
        maxWidth: 100,
      },
      {
        Header: 'Queue',
        accessor: 'queue',
        maxWidth: 100,
      },
      {
        id: 'startedTimeString',
        Header: 'Start Time',
        maxWidth: 100,
        accessor: (d) => { /* console.log(d.startedTime); */ return utils.timeConverter(d.startedTime);} // Custom value accessors!
      },
      {
        id: 'finishedTimeString',
        Header: 'Finish Time',
        maxWidth: 100,
        accessor: (d) => { return utils.timeConverter(d.finishedTime);} // Custom value accessors!
      },
      {
        Header: 'State',
        accessor: 'state',
        maxWidth: 100,
      },
      {
        Header: 'Detailed View',
        maxWidth: 100,
        Cell: (props) => {
          if(props.original.state === "FINISHED"){
            return ( <Button primary onClick={()=>{this.props.viewHistory(props.original.id,utils.timeConverter2(props.original.startedTime))}}>View</Button> )
          }else if(props.original.state === "RUNNING" || props.original.state === "ACCEPTED" ){
            return ( <Button primary disabled>View</Button> ) // 190320 je.kim 로딩중 버튼 대신 disabled 버튼으로 처리
          }else{
            return ( <Button primary disabled>View</Button> )
          }

        } // Custom cell components!
      }
    ]


    return(
      <div style={{width:'100%'}}>
        <h4>Total : {data_count}</h4>
        <ReactTable
          data={data}
          columns={columns}
          minRows={10}
          defaultPageSize={10}
          defaultSorted={[
            {
              id: "startedTimeString",
              desc: true
            }
          ]}
        />
      </div>
    )
  }

}

export default AppHistory;
