import React, { Component } from 'react';
import _ from 'lodash';
import { Checkbox, Button } from 'semantic-ui-react'
import ReactTooltip from 'react-tooltip'
import './Select.css';

class Select extends Component{
  constructor() {
    super();
    this.state = {
      showMenu: false,
    };
    this.showMenu = this.showMenu.bind(this);
    this.closeMenu = this.closeMenu.bind(this);
  }

  showMenu(event) {
    event.preventDefault();
    this.setState({ showMenu: true }, () => {
      document.addEventListener('click', this.closeMenu);
    });
  }

  closeMenu(event) {
    if (!this.dropdownMenu.contains(event.target)) {
      this.setState({ showMenu: false }, () => {
        document.removeEventListener('click', this.closeMenu);
      });

    }
  }

  render() {
    // console.log(this.props.data);
    let {data,viewOrHideTraces} = this.props;
    const menuStyle = {
      display : Object.keys(data).length > 0 ? "show" : "none"
    }
    const ContainerChecked = _.map(data,(container_information,container_id)=>{
      return(
        <div
          className='item'
          key={container_information.index}
          >
          <div className='ui input'>
            <Checkbox
              label={
                <label data-for='enrich' data-tip={container_information.node}>
                  [PID {container_information.pid}]  {container_id}
                </label>
              }
              defaultChecked
              onChange={ (e, checked) => { e.stopPropagation(); /* onToggle 이 실행되지 않도록 함 */  viewOrHideTraces(checked,container_id);  }  }
              />
          </div>
        </div>
      )
    })

    return(
      <div>
        <div>

          <Button content='Show Containers' icon='angle down' labelPosition='right' onClick={this.showMenu}/>
          {
            (() => {
              if (this.state.showMenu) {
                return (
                  <div
                    className="ui vertical menu SelectMeun"
                    ref={(element) => {
                      this.dropdownMenu = element;
                    }}
                    style={menuStyle}
                  >
                    {ContainerChecked}
                  </div>
                );
              } else {
                return null
              }
            })()
          }
        </div>
        <ReactTooltip id='enrich' getContent={(dataTip) => `${dataTip}`}/>
      </div>
    );
  }
}


export default Select;
