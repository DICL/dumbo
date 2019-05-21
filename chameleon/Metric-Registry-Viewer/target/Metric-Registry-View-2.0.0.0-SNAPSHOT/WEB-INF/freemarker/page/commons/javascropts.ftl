<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<% 
	String[] page_url = request.getRequestURL().toString().split("/");
	String pageName = page_url[page_url.length - 1].replaceAll(".jsp", "");
%>
<script>
var contextPath = "${contextPath}";
</script>

<script type="text/javascript" src="${contextPath}/js/common.js"></script>
<script type="text/javascript" src="${contextPath}/js/Handlebars.custom.js"></script>
<script type="text/javascript" src="${contextPath}/js/operationsRunning.js"></script>
<script type="text/javascript" src="${contextPath}/js/page/${pageName }.js"></script>

<%-- <script type="text/javascript" src="${contextPath}/js/index.js"></script> --%>
