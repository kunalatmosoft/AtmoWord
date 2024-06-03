const toolbarOptions = [
    ['bold', 'italic', 'underline', 'strike'],        // toggled buttons
    ['blockquote', 'code-block'],
    ['link', 'image', 'video', 'formula'],
   // [['code-block']] 
    [{ 'header': 1 }, { 'header': 2 }],               // custom button values
    [{ 'list': 'ordered'}, { 'list': 'bullet' }, { 'list': 'check' }],
    [{ 'script': 'sub'}, { 'script': 'super' }],      // superscript/subscript
    [{ 'indent': '-1'}, { 'indent': '+1' }],          // outdent/indent
    [{ 'direction': 'rtl' }],                         // text direction
  
    [{ 'size': ['small', false, 'large', 'huge'] }],  // custom dropdown
    [{ 'header': [1, 2, 3, 4, 5, 6, false] }],
  
    [{ 'color': [] }, { 'background': [] }],          // dropdown with defaults from theme
    [{ 'font': [] }],
    [{ 'align': [] }],
  
    ['clean']                                         // remove formatting button
  ];
  
  const quill = new Quill('#editor', {
    modules: {
         syntax: true,              // Include syntax module
      toolbar: toolbarOptions
    },
    theme: 'snow'
  });

// Function to trigger confetti animation
function triggerConfetti() {
    var duration = 3 * 1000;
    var animationEnd = Date.now() + duration;
    var defaults = {
        startVelocity: 30,
        spread: 360,
        ticks: 60,
        zIndex: 0,
        colors: ['#ff0a54', '#ff477e', '#ff7096', '#ff85a1', '#fbb1bd', '#f9bec7', '#f7cad0', '#f4bec1', '#f1a6b5', '#ee8ca8']
    };

    function randomInRange(min, max) {
        return Math.random() * (max - min) + min;
    }

    var interval = setInterval(function() {
        var timeLeft = animationEnd - Date.now();

        if (timeLeft <= 0) {
            return clearInterval(interval);
        }

        var particleCount = 100 * (timeLeft / duration);
        // since particles fall down, start a bit higher than random
        confetti(Object.assign({}, defaults, {
            particleCount,
            origin: {
                x: randomInRange(0.1, 0.3),
                y: Math.random() - 0.2
            }
        }));
        confetti(Object.assign({}, defaults, {
            particleCount,
            origin: {
                x: randomInRange(0.7, 0.9),
                y: Math.random() - 0.2
            }
        }));
    }, 250);
}


  document.getElementById('downloadBtn').addEventListener('click', function() {
    // Get the plain text from the Quill editor
triggerConfetti();
    var editorContent = quill.getText();
    
    // Create a blob of the plain text
    var blob = new Blob([editorContent], { type: 'text/plain' });
    
    // Create a link element
    var a = document.createElement('a');
    
    // Create a URL for the blob and set it as the href attribute
    var url = URL.createObjectURL(blob);
    a.href = url;
    
    // Set the download attribute with a filename
    a.download = 'content.txt';
    
    // Append the link to the body
    document.body.appendChild(a);
    
    // Programmatically click the link to trigger the download
    a.click();
    
    // Remove the link from the document
    document.body.removeChild(a);
});
/*   const Delta = Quill.import('delta');
quill.setContents(
  new Delta()
    .insert('const language = "JavaScript";')
    .insert('\n', { 'code-block': 'javascript' })
    .insert('console.log("I love " + language + "!");')
    .insert('\n', { 'code-block': 'javascript' })
); */