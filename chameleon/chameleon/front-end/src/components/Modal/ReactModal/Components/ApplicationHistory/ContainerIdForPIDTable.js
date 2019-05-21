import React, { Component } from 'react';
import ReactTable from 'react-table'
import 'react-table/react-table.css'



const ContainerIdForPIDTable = ( {container_list} ) =>{
  // console.log(container_list);
  const data = container_list;
  const data_count = container_list.length
  const columns = [
    {
      Header: 'Container ID',
      accessor: 'container_id',
    },
    {
      Header: 'PID',
      accessor: 'pid',
    },
  ]

  return(
    <div id="containerid-for-pid-list">
      <ReactTable
        data={data}
        columns={columns}
        minRows={3}
        defaultPageSize={3}
      />
    </div>
  )
};

export default ContainerIdForPIDTable
