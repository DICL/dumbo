import React, { Component } from 'react';
import './DropDown.css'

class DropDown extends Component{
  static defaultProps = {
    list: [
      'Menu item 1',
      'Menu item 2',
      'Menu item 3',
      'Menu item 4',
    ]
  }

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
    const dropdown_list = this.props.list.map((item,index)=>{
      return(
        <div
          className='item'
          key={index}
          >
          <div className='ui input'>
            <button>{item}</button>
          </div>
        </div>
      )
    })
    return (
      <div>
        <a onClick={this.showMenu}>
          Show Cotainers
        </a>
        {
          this.state.showMenu
            ? (
              <div
                className="ui vertical menu DropDown"
                ref={(element) => {
                  this.dropdownMenu = element;
                }}
              >
                {dropdown_list}
              </div>
            )
            : (
              null
            )
        }
      </div>
    );
  }
}



export default DropDown;
