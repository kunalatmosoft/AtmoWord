A popular JavaScript animation library that can add celebratory effects like confetti to web pages is **Confetti.js**. This lightweight library allows you to create fun and dynamic confetti animations with ease.

Here's a simple example of how to use Confetti.js to sprinkle a celebration effect:

1. **Include the Confetti.js library in your project**:
   You can either download the library or include it directly from a CDN.

   ```html
   <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
   ```

2. **Add a script to trigger the confetti effect**:
   You can trigger the confetti animation based on user actions, such as a button click.

   ```html
   <button onclick="confetti()">Celebrate 🎉</button>

   <script>
     function confetti() {
       confetti({
         particleCount: 100,
         spread: 70,
         origin: { y: 0.6 }
       });
     }
   </script>
   ```

This will create a simple confetti effect when the button is clicked. You can customize the `confetti` function parameters to adjust the number of particles, spread, and other properties to suit your celebration needs.

Confetti.js is easy to use and can add a festive touch to your web applications, making it a great choice for celebratory animations.