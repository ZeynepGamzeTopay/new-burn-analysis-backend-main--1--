using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using BurnAnalysisApp.Models;
using BurnAnalysisApp.Data;
using System;
using System.IO;
using System.Threading.Tasks;

namespace BurnAnalysisApp.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class BurnFormController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly IWebHostEnvironment _env;

        public BurnFormController(AppDbContext context, IWebHostEnvironment env)
        {
            _context = context;
            _env = env;
        }

        [HttpPost]
        public async Task<IActionResult> PostBurnForm([FromForm] BurnFormInfo burnForm, [FromForm] IFormFile? photo)
        {
            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            // Fotoğraf işleme
            string? photoPath = null;
            if (photo != null)
            {
                var uploadDir = Path.Combine(_env.WebRootPath, "uploads");

                // Klasör var mı kontrolü yapma (mevcut olduğunu varsayıyoruz)
                if (!Directory.Exists(uploadDir))
                {
                    Directory.CreateDirectory(uploadDir); // Eğer klasör yoksa oluştur
                }

                // Fotoğraf dosyasının adını benzersiz yapma
                var fileName = $"{Guid.NewGuid()}{Path.GetExtension(photo.FileName)}";
                photoPath = Path.Combine(uploadDir, fileName);

                // Fotoğrafı sunucuya kaydetme
                using (var stream = new FileStream(photoPath, FileMode.Create))
                {
                    await photo.CopyToAsync(stream);
                }

                // Fotoğrafın göreceli yolunu al
                photoPath = $"/uploads/{fileName}";
            }

            try
            {
                // Hasta kaydını kontrol et
                var existingPatient = await _context.Patients
                    .FirstOrDefaultAsync(p => p.Name == burnForm.PatientName &&
                                               p.Age == burnForm.Age &&
                                               p.Email== burnForm.Email&& 
                                               p.BurnArea== burnForm.BurnArea&&
                                               p.Gender == burnForm.Gender);

                if (existingPatient != null)
                {
                    // Mevcut hasta kaydını güncelle
                    existingPatient.BurnCause = burnForm.BurnCause;
                    existingPatient.HospitalArrivalDate = burnForm.HospitalArrivalDate;
                    existingPatient.BurnOccurrenceDate = burnForm.BurnOccurrenceDate;

                    // Fotoğraf path güncellemesi yapılacaksa
                    if (!string.IsNullOrEmpty(photoPath))
                    {
                        existingPatient.PhotoPath = photoPath;
                    }
                }
                else
                {
                    // Yeni hasta kaydı oluştur
                    var newPatient = new PatientInfo
                    {
                        Name = burnForm.PatientName,
                        Age = burnForm.Age,
                        Email = burnForm.Email,
                        Gender = burnForm.Gender,
                        BurnCause = burnForm.BurnCause,
                        BurnArea = burnForm.BurnArea,
                        HospitalArrivalDate = burnForm.HospitalArrivalDate,
                        BurnOccurrenceDate = burnForm.BurnOccurrenceDate,
                        PhotoPath = photoPath
                    };
                    _context.Patients.Add(newPatient);
                }

                await _context.SaveChangesAsync();
                return Ok(new { Message = "Form successfully processed." });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { Message = "An error occurred.", Error = ex.Message });
            }
        }
    }
}
