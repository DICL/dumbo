<%@ page contentType="text/html; charset=UTF-8" %>
<!DOCTYPE html>
<html>

<jsp:include page="commons/head.jsp" flush="true"></jsp:include>


<body>
	<div id="root">
		<div class="container">
			<div class="row">
			  	<jsp:include page="commons/header.jsp" flush="true"></jsp:include>
			  	
				<div class="col-12 mt-5">
					<div class="col-12 px-3 py-3 border">
						<h2>Add metric</h2>
						<div class="col-12 px-3 py-3 ">
							<div class="row mb-3">
								<div class="col-3">
									<label>Name :</label>
								</div>
								<div class="col-9 text-left">
									<input name="name" id="name" type="text" class="form-control"  value=""/>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-3">
									<label>Description :</label>
								</div>
								<div class="col-9 text-left">
									<input name="description" id="description" type="text" class="form-control" value=""/>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-3">
									<label>PID Symbol :</label>
								</div>
								<div class="col-9 text-left">
									<input name="pid_symbol" id="pid_symbol" type="text" class="form-control" value=""/>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-3">
									<label>Y-axis Label :</label>
								</div>
								<div class="col-9 text-left">
									<input name="y_axis_label" id="y_axis_label" type="text" class="form-control" value=""/>
								</div>
							</div>
							<div class="row">
								<div class="col-12">
									<label>Parser script:</label>
								</div>
								<div class="col-12">
									<textarea name="parser_script" id="parser_script" rows="15" cols="" class="form-control"  style="width: 100%"></textarea>
								</div>
							</div>
							
						</div>
						<div class="col-12 px-3 py-3 text-right">
							<button type="button" id="register_metric" class="btn btn-primary">Register</button>
							<a href="${contextPath}/" class="btn btn-primary">Cancel</a>
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