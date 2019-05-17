/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.xiilab.lustre.page;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import javax.servlet.ServletContext;
import javax.servlet.http.HttpServletRequest;


@Controller
public class PageController {
	@Autowired
    private ServletContext servletContext;
	
	@RequestMapping(value = "/", method = RequestMethod.GET)
	public String MDS_Setting_page(ModelMap model, HttpServletRequest request) {
	    model.addAttribute("contextPath", servletContext.getContextPath());
	    model.addAttribute("pageName", "MDS_Setting");
//	    return "page/MDS_Setting";
	    return "page/index";
	}
	
	@RequestMapping(value = "/OSS_Setting", method = RequestMethod.GET)
	public String OSS_Setting_page(ModelMap model, HttpServletRequest request) {
	    model.addAttribute("contextPath", servletContext.getContextPath());
	    model.addAttribute("pageName", "OSS_Setting");
	    return "page/OSS_Setting";
	}
	@RequestMapping(value = "/Client_Setting", method = RequestMethod.GET)
	public String Client_Setting_page(ModelMap model, HttpServletRequest request) {
		model.addAttribute("contextPath", servletContext.getContextPath());
		model.addAttribute("pageName", "Client_Setting");
		return "page/Client_Setting";
	}
	@RequestMapping(value = "/LNET_Setting", method = RequestMethod.GET)
	public String LNET_Setting_page(ModelMap model, HttpServletRequest request) {
		model.addAttribute("contextPath", servletContext.getContextPath());
		model.addAttribute("pageName", "LNET_Setting");
		return "page/LNET_Setting";
	}
	@RequestMapping(value = "/Backup", method = RequestMethod.GET)
	public String Backup_page(ModelMap model, HttpServletRequest request) {
		model.addAttribute("contextPath", servletContext.getContextPath());
		model.addAttribute("pageName", "Backup");
		return "page/Backup";
	}
	@RequestMapping(value = "/Restore", method = RequestMethod.GET)
	public String Restore_page(ModelMap model, HttpServletRequest request) {
		model.addAttribute("contextPath", servletContext.getContextPath());
		model.addAttribute("pageName", "Restore");
		return "page/Restore";
	}
	
}

