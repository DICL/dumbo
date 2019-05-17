<%@page import="org.springframework.context.annotation.Import"%>
<%@ page contentType="text/html; charset=UTF-8" %>
<%@ page import="com.xiilab.metric.model.MetricRegistryVO"%>
<%@ page import="java.util.List"%>
<%-- <%@taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
 --%>
<%-- JSTL 이 안되어 ajax 이용 --%>


<!DOCTYPE html>
<html>

<jsp:include page="commons/head.jsp" flush="true"></jsp:include>

<%-- <script type="text/javascript" src="${contextPath}/js/page/Metric_Registry_View.js"></script> --%>

<body>
	<div id="root">
		<div class="container">
			<div class="row">
			  	<jsp:include page="commons/header.jsp" flush="true"></jsp:include>
			  	
			  	<!-- menu -->
				<div class="col-12">
					<div class="form-inline">
						<h2>Metric Registry View</h2>
					</div>
				</div>
			  	
				<div class="col-12 mt-1">
					<div class="col-12">
						<h5>Registered metrics:</h5>
						<div id="metric_list" class="col-12 px-3 py-3 border">
							<!-- <div class="row mb-3">
								<div class="col-6">
									pidstat.cpu
								</div>
								<div class="col-6 text-right">
									<button type="button" id="mds-apply" class="btn btn-primary">Del</button>
									<button type="button" id="mds-reset" class="btn btn-primary">View</button>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-6">
									pidstat.mem
								</div>
								<div class="col-6 text-right">
									<button type="button" id="mds-apply" class="btn btn-primary">Del</button>
									<button type="button" id="mds-reset" class="btn btn-primary">View</button>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-6">
									pidstat.diskio.read
								</div>
								<div class="col-6 text-right">
									<button type="button" id="mds-apply" class="btn btn-primary">Del</button>
									<button type="button" id="mds-reset" class="btn btn-primary">View</button>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-6">
									pidstat.diskio.write
								</div>
								<div class="col-6 text-right">
									<button type="button" id="mds-apply" class="btn btn-primary">Del</button>
									<button type="button" id="mds-reset" class="btn btn-primary">View</button>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-6">
									perf.branch-misses
								</div>
								<div class="col-6 text-right">
									<button type="button" id="mds-apply" class="btn btn-primary">Del</button>
									<button type="button" id="mds-reset" class="btn btn-primary">View</button>
								</div> -->
							</div>
						</div>
						<div class="col-12 px-3 py-3 text-right">
							<!-- <button type="button" id="" class="btn btn-primary">Add</button> -->
							<a href="${contextPath}/addMetric" class="btn btn-primary">Add</a>
						</div>
					</div>
					
				</div>
			</div>
		</div>
	</div>
</body>

<jsp:include page="commons/javascropts.jsp" flush="true"></jsp:include>
<jsp:include page="commons/modals.jsp" flush="true"></jsp:include>
</html>