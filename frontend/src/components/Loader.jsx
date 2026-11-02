// Loading indicator component.
function Loader() {
  return (
    <div className="loaderOverlay">
      <div className="spinner"></div>

      <h3>Processing Satellite Image</h3>

      <p>
        Preparing the image and running the reconstruction
        pipeline...
      </p>
    </div>
  );
}

export default Loader;