<%@ page contentType="text/html; charset=UTF-8" %>


<!-- loading bar -->
<div class="pure-loarding">
	<div class="lds-spinner">
		<div></div>
		<div></div>
		<div></div>
		<div></div>
		<div></div>
		<div></div>
		<div></div>
		<div></div>
		<div></div>
		<div></div>
		<div></div>
		<div></div>
	</div>
	<p class="message">
		test
	</p>
</div>

<!-- menu -->
<div class="col-12">
	<div class="form-inline">
		<h2>LustreManager View</h2>
		<button type="button" class="btn btn-primary btn-sm ml-3 operationsRunning" data-target=".bd-managerque-modal-lg">Operations Running</button>
	</div>
	
	<div class="col-12 border px-3 py-3">
		<ul class="nav nav-pills nav-fill">
			<li class="nav-item"><a id="MDS" class="nav-link" href="${contextPath}/">MDS Setting</a></li>
			<li class="nav-item"><a id="OSS_Setting" class="nav-link" href="${contextPath}/OSS_Setting">OSS Setting</a></li>
			<li class="nav-item"><a id="Client_Setting" class="nav-link" href="${contextPath}/Client_Setting">Client Setting</a></li>
			<li class="nav-item"><a id="LNET_Setting" class="nav-link" href="${contextPath}/LNET_Setting">LNET Setting</a></li>
			<li class="nav-item"><a id="Backup" class="nav-link" href="${contextPath}/Backup">Backup</a></li>
			<li class="nav-item"><a id="Restore" class="nav-link" href="${contextPath}/Restore">Restore</a></li>
		</ul>
	</div>
</div>

<script>
$(function () {
$('.nav-item a[href="'+window.location.pathname+'"]').addClass('active'); //** menu active
})
</script>