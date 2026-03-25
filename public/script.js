document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("movieForm");

    if (form) {
        form.addEventListener("submit", () => {
            const button = form.querySelector("button[type='submit']");
            button.innerHTML = "⏳ Generating Recommendations...";
            button.disabled = true;
            button.classList.add("opacity-70", "cursor-not-allowed");
        });
    }
});