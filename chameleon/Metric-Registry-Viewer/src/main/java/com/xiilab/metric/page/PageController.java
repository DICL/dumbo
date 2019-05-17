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

package com.xiilab.metric.page;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import com.xiilab.metric.api.MetricService;

import javax.servlet.ServletContext;
import javax.servlet.http.HttpServletRequest;


@Controller
public class PageController {
	@Autowired
    private ServletContext servletContext;
	@Autowired
	private MetricService metricService;
	
	@RequestMapping(value = "/", method = RequestMethod.GET)
	public String Metric_Registry_View(ModelMap model, HttpServletRequest request) {
	    model.addAttribute("contextPath", servletContext.getContextPath());
	    model.addAttribute("pageName", "Metric_Registry_View");
	    return "page/Metric_Registry_View";
	}
	
	@RequestMapping(value = "/addMetric", method = RequestMethod.GET)
	public String Add_Metric(ModelMap model, HttpServletRequest request) {
	    model.addAttribute("contextPath", servletContext.getContextPath());
	    model.addAttribute("pageName", "Add_Metric");
	    return "page/Add_Metric";
	}
	
	@RequestMapping(value = "/metric/{num}", method = RequestMethod.GET)
	public String View_Metric(ModelMap model, HttpServletRequest request, @PathVariable Long num) {
		model.addAttribute("contextPath", servletContext.getContextPath());
		model.addAttribute("metric", metricService.View_Metric(num));
		model.addAttribute("pageName", "View_Metric");
		return "page/View_Metric";
	}
	
}

