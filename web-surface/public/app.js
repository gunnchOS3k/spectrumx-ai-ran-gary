const link = document.getElementById("portal-return");
fetch("/project-surface.json")
  .then((res) => res.json())
  .then((data) => {
    if (data && typeof data.returnUrl === "string" && /^https:\/\//.test(data.returnUrl)) {
      link.href = data.returnUrl;
    }
  })
  .catch(() => {});
